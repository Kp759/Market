"""
data_ingestion.py — async data-ingestion layer for the stock-analysis system.

Responsibilities
----------------
* Fetch OHLCV price history and fundamental ratios (P/E, EPS, revenue) via
  yfinance (runs in a thread-pool to avoid blocking the event loop).
* Fetch recent news articles per ticker from NewsAPI using httpx async calls.
* Cache every result to SQLite (aiosqlite) with configurable TTLs:
    - prices / fundamentals : 15 minutes
    - news articles         : 1 hour
* Expose a single entry-point:
    DataIngestion.get_stock_data(tickers) -> dict

Schema (see db/schema.sql)
--------------------------
    prices          (ticker, data JSON, fetched_at, ttl_expires_at)
    fundamentals    (ticker, pe_ratio, eps, revenue, data JSON, fetched_at, ttl_expires_at)
    news_articles   (ticker, data JSON, fetched_at, ttl_expires_at)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiosqlite
import httpx
import yfinance as yf

from config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA_PATH = __file__.replace("data_ingestion.py", "db/schema.sql")


async def _ensure_schema(db: aiosqlite.Connection) -> None:
    """Create tables if they don't exist yet."""
    with open(_SCHEMA_PATH) as fh:
        sql = fh.read()
    await db.executescript(sql)
    await db.commit()


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# DataIngestion
# ---------------------------------------------------------------------------

class DataIngestion:
    """
    Async data-ingestion class.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Defaults to settings.db_path.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.db_path
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DataIngestion":
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await _ensure_schema(self._db)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_stock_data(self, tickers: list[str]) -> dict[str, Any]:
        """
        Fetch prices, fundamentals, and news for every ticker in *tickers*.

        Returns a dict keyed by ticker::

            {
              "AAPL": {
                "prices":       [...],   # list of OHLCV dicts
                "fundamentals": {...},   # P/E, EPS, revenue + raw blob
                "news":         [...],   # list of article dicts
              },
              ...
            }
        """
        if self._db is None:
            raise RuntimeError(
                "DataIngestion must be used as an async context manager."
            )

        results = await asyncio.gather(
            *[self._fetch_ticker(ticker) for ticker in tickers],
            return_exceptions=True,
        )

        output: dict[str, Any] = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                output[ticker] = {"error": str(result)}
            else:
                output[ticker] = result
        return output

    # ------------------------------------------------------------------
    # Per-ticker orchestration
    # ------------------------------------------------------------------

    async def _fetch_ticker(self, ticker: str) -> dict[str, Any]:
        prices_task       = asyncio.create_task(self._get_prices(ticker))
        fundamentals_task = asyncio.create_task(self._get_fundamentals(ticker))
        news_task         = asyncio.create_task(self._get_news(ticker))

        prices, fundamentals, news = await asyncio.gather(
            prices_task, fundamentals_task, news_task,
            return_exceptions=True,
        )

        return {
            "prices":       prices       if not isinstance(prices,       Exception) else {"error": str(prices)},
            "fundamentals": fundamentals if not isinstance(fundamentals, Exception) else {"error": str(fundamentals)},
            "news":         news         if not isinstance(news,         Exception) else {"error": str(news)},
        }

    # ------------------------------------------------------------------
    # Price data (yfinance)
    # ------------------------------------------------------------------

    async def _get_prices(self, ticker: str) -> list[dict[str, Any]]:
        cached = await self._cache_get("prices", ticker)
        if cached is not None:
            return cached

        data = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_prices_sync, ticker
        )
        await self._cache_set("prices", ticker, data, settings.price_ttl)
        return data

    @staticmethod
    def _fetch_prices_sync(ticker: str) -> list[dict[str, Any]]:
        tf = yf.Ticker(ticker)
        hist = tf.history(period="1mo", interval="1d")
        records = []
        for ts, row in hist.iterrows():
            records.append({
                "date":   ts.isoformat(),
                "open":   round(float(row["Open"]),   4),
                "high":   round(float(row["High"]),   4),
                "low":    round(float(row["Low"]),    4),
                "close":  round(float(row["Close"]),  4),
                "volume": int(row["Volume"]),
            })
        return records

    # ------------------------------------------------------------------
    # Fundamental data (yfinance)
    # ------------------------------------------------------------------

    async def _get_fundamentals(self, ticker: str) -> dict[str, Any]:
        cached = await self._cache_get("fundamentals", ticker)
        if cached is not None:
            return cached

        data = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_fundamentals_sync, ticker
        )
        await self._cache_set("fundamentals", ticker, data, settings.price_ttl)
        return data

    @staticmethod
    def _fetch_fundamentals_sync(ticker: str) -> dict[str, Any]:
        tf = yf.Ticker(ticker)
        info = tf.info or {}

        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        eps      = info.get("trailingEps") or info.get("forwardEps")
        revenue  = info.get("totalRevenue")

        return {
            "pe_ratio":         pe_ratio,
            "eps":              eps,
            "revenue":          revenue,
            "market_cap":       info.get("marketCap"),
            "dividend_yield":   info.get("dividendYield"),
            "52w_high":         info.get("fiftyTwoWeekHigh"),
            "52w_low":          info.get("fiftyTwoWeekLow"),
            "beta":             info.get("beta"),
            "sector":           info.get("sector"),
            "industry":         info.get("industry"),
        }

    # ------------------------------------------------------------------
    # News (NewsAPI via httpx async)
    # ------------------------------------------------------------------

    async def _get_news(self, ticker: str) -> list[dict[str, Any]]:
        cached = await self._cache_get("news_articles", ticker)
        if cached is not None:
            return cached

        if not settings.news_api_key:
            return []

        articles = await self._fetch_news_async(ticker)
        await self._cache_set("news_articles", ticker, articles, settings.news_ttl)
        return articles

    async def _fetch_news_async(self, ticker: str) -> list[dict[str, Any]]:
        url = f"{settings.newsapi_base_url}/everything"
        params = {
            "q":        ticker,
            "sortBy":   "publishedAt",
            "pageSize": settings.newsapi_page_size,
            "language": "en",
            "apiKey":   settings.news_api_key,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        articles = []
        for article in payload.get("articles", []):
            articles.append({
                "title":       article.get("title"),
                "description": article.get("description"),
                "url":         article.get("url"),
                "source":      article.get("source", {}).get("name"),
                "published_at":article.get("publishedAt"),
            })
        return articles

    # ------------------------------------------------------------------
    # SQLite cache helpers
    # ------------------------------------------------------------------

    async def _cache_get(
        self, table: str, ticker: str
    ) -> list[Any] | dict[str, Any] | None:
        """Return cached data if a fresh (non-expired) row exists, else None."""
        now = _now()
        async with self._db.execute(
            f"SELECT data FROM {table} WHERE ticker = ? AND ttl_expires_at > ? "
            f"ORDER BY fetched_at DESC LIMIT 1",
            (ticker, now),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

    async def _cache_set(
        self, table: str, ticker: str, data: Any, ttl: int
    ) -> None:
        """Insert a fresh cache row; prune expired rows for this ticker."""
        now        = _now()
        expires_at = now + ttl
        blob       = json.dumps(data)

        # Prune stale rows first to keep the table lean
        await self._db.execute(
            f"DELETE FROM {table} WHERE ticker = ? AND ttl_expires_at <= ?",
            (ticker, now),
        )
        await self._db.execute(
            f"INSERT INTO {table} (ticker, data, fetched_at, ttl_expires_at) "
            f"VALUES (?, ?, ?, ?)",
            (ticker, blob, now, expires_at),
        )
        await self._db.commit()


# ---------------------------------------------------------------------------
# __main__ — quick smoke-test
# ---------------------------------------------------------------------------

async def _main() -> None:
    tickers = ["AAPL", "NVDA", "MSFT"]

    async with DataIngestion() as ingestion:
        print(f"Fetching data for {tickers} …")
        result = await ingestion.get_stock_data(tickers)

    for ticker, data in result.items():
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        prices = data.get("prices", [])
        if isinstance(prices, list) and prices:
            latest = prices[-1]
            print(f"  Latest close : {latest['close']}  ({latest['date'][:10]})")
            print(f"  OHLCV rows   : {len(prices)}")
        else:
            print(f"  Prices       : {prices}")

        fundamentals = data.get("fundamentals", {})
        if isinstance(fundamentals, dict):
            print(f"  P/E ratio    : {fundamentals.get('pe_ratio')}")
            print(f"  EPS          : {fundamentals.get('eps')}")
            print(f"  Revenue      : {fundamentals.get('revenue')}")
        else:
            print(f"  Fundamentals : {fundamentals}")

        news = data.get("news", [])
        if isinstance(news, list) and news:
            print(f"  News articles: {len(news)}")
            for article in news[:2]:
                print(f"    - {article.get('title', 'n/a')[:80]}")
        else:
            print(f"  News         : {news or 'no API key configured'}")


if __name__ == "__main__":
    asyncio.run(_main())
