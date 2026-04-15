"""
data_ingestion.py — async data-ingestion layer for the stock-analysis system.

Responsibilities
----------------
* get_ohlcv(tickers)       — 1-year daily OHLCV via yfinance (thread-pool)
* get_fundamentals(tickers) — P/E, EPS, revenue growth, market cap via yfinance
* get_news(tickers)        — last 7 days of articles via NewsAPI (httpx async)
* get_options_iv(tickers)  — ATM implied volatility from nearest-expiry options chain
* get_stock_data(tickers)  — unified method combining all four above
* All results cached to SQLite (aiosqlite) with configurable TTLs

Cache TTLs
----------
    prices / fundamentals : 15 minutes
    news / IV             : 1 hour

Schema: see db/schema.sql
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import numpy as np
import yfinance as yf

from config import settings

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent / "db" / "schema.sql"


async def _ensure_schema(db: aiosqlite.Connection) -> None:
    """Create tables from schema.sql if they don't exist yet."""
    sql = _SCHEMA_PATH.read_text()
    await db.executescript(sql)
    await db.commit()


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# DataIngestion
# ---------------------------------------------------------------------------


class DataIngestion:
    """
    Async data-ingestion class for OHLCV, fundamentals, news, and options IV.

    Must be used as an async context manager::

        async with DataIngestion() as ing:
            data = await ing.get_stock_data(["AAPL", "NVDA"])

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database.  Defaults to ``settings.db_path``.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.db_path
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Context manager
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

    # ==================================================================
    # Public methods
    # ==================================================================

    async def get_stock_data(self, tickers: list[str]) -> dict[str, Any]:
        """
        Fetch OHLCV, fundamentals, news, and options IV for every ticker.

        Tickers are processed in parallel via ``asyncio.gather``.

        Returns
        -------
        dict
            ``{ticker: {prices, fundamentals, news, options_iv}}``
        """
        self._require_db()
        results = await asyncio.gather(
            *[self._fetch_all_for_ticker(t) for t in tickers],
            return_exceptions=True,
        )
        return {
            ticker: result if not isinstance(result, Exception) else {"error": str(result)}
            for ticker, result in zip(tickers, results)
        }

    async def get_ohlcv(self, tickers: list[str]) -> dict[str, list[dict]]:
        """
        Return 1-year daily OHLCV records for each ticker.

        Returns
        -------
        dict
            ``{ticker: [{date, open, high, low, close, volume}, ...]}``
        """
        self._require_db()
        results = await asyncio.gather(
            *[self._get_prices(t) for t in tickers], return_exceptions=True
        )
        return {
            t: r if not isinstance(r, Exception) else []
            for t, r in zip(tickers, results)
        }

    async def get_fundamentals(self, tickers: list[str]) -> dict[str, dict]:
        """
        Return fundamental ratios (P/E, EPS, revenue growth, market cap) per ticker.

        Returns
        -------
        dict
            ``{ticker: {pe_ratio, eps, revenue_growth, market_cap, ...}}``
        """
        self._require_db()
        results = await asyncio.gather(
            *[self._get_fundamentals(t) for t in tickers], return_exceptions=True
        )
        return {
            t: r if not isinstance(r, Exception) else {}
            for t, r in zip(tickers, results)
        }

    async def get_news(self, tickers: list[str]) -> dict[str, list[dict]]:
        """
        Return the last 7 days of news articles per ticker via NewsAPI.

        Returns
        -------
        dict
            ``{ticker: [{title, description, url, source, published_at}, ...]}``
        """
        self._require_db()
        results = await asyncio.gather(
            *[self._get_news(t) for t in tickers], return_exceptions=True
        )
        return {
            t: r if not isinstance(r, Exception) else []
            for t, r in zip(tickers, results)
        }

    async def get_options_iv(self, tickers: list[str]) -> dict[str, float | None]:
        """
        Return ATM implied volatility from the nearest-expiry options chain.

        Queries ``yfinance`` in a thread-pool executor to avoid blocking.

        Returns
        -------
        dict
            ``{ticker: iv_float_or_None}``
        """
        self._require_db()
        results = await asyncio.gather(
            *[self._get_iv(t) for t in tickers], return_exceptions=True
        )
        return {
            t: r if not isinstance(r, Exception) else None
            for t, r in zip(tickers, results)
        }

    # ==================================================================
    # Per-ticker orchestration (private)
    # ==================================================================

    async def _fetch_all_for_ticker(self, ticker: str) -> dict[str, Any]:
        prices, fundamentals, news, iv = await asyncio.gather(
            self._get_prices(ticker),
            self._get_fundamentals(ticker),
            self._get_news(ticker),
            self._get_iv(ticker),
            return_exceptions=True,
        )
        return {
            "prices":      prices       if not isinstance(prices,       Exception) else [],
            "fundamentals":fundamentals if not isinstance(fundamentals, Exception) else {},
            "news":        news         if not isinstance(news,         Exception) else [],
            "options_iv":  iv           if not isinstance(iv,           Exception) else None,
        }

    # ==================================================================
    # OHLCV — 1 year daily
    # ==================================================================

    async def _get_prices(self, ticker: str) -> list[dict[str, Any]]:
        cached = await self._cache_get("prices", ticker)
        if cached is not None:
            return cached
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_prices_sync, ticker)
        await self._cache_set("prices", ticker, data, settings.price_ttl)
        return data

    @staticmethod
    def _fetch_prices_sync(ticker: str) -> list[dict[str, Any]]:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d")
        records = []
        for ts, row in hist.iterrows():
            records.append(
                {
                    "date":   ts.isoformat(),
                    "open":   round(float(row["Open"]),  4),
                    "high":   round(float(row["High"]),  4),
                    "low":    round(float(row["Low"]),   4),
                    "close":  round(float(row["Close"]), 4),
                    "volume": int(row["Volume"]),
                }
            )
        return records

    # ==================================================================
    # Fundamentals
    # ==================================================================

    async def _get_fundamentals(self, ticker: str) -> dict[str, Any]:
        cached = await self._cache_get("fundamentals", ticker)
        if cached is not None:
            return cached
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_fundamentals_sync, ticker)
        await self._cache_set("fundamentals", ticker, data, settings.price_ttl)
        return data

    @staticmethod
    def _fetch_fundamentals_sync(ticker: str) -> dict[str, Any]:
        info = yf.Ticker(ticker).info or {}

        # Revenue growth: prefer quarterly, fall back to annual
        revenue_growth = info.get("revenueGrowth") or info.get("earningsGrowth")

        return {
            "pe_ratio":       info.get("trailingPE") or info.get("forwardPE"),
            "eps":            info.get("trailingEps") or info.get("forwardEps"),
            "eps_growth":     info.get("earningsGrowth"),
            "revenue":        info.get("totalRevenue"),
            "revenue_growth": revenue_growth,
            "market_cap":     info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "beta":           info.get("beta"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
            "sector":         info.get("sector"),
            "industry":       info.get("industry"),
        }

    # ==================================================================
    # News — last 7 days via NewsAPI
    # ==================================================================

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
        from_date = (
            datetime.now(timezone.utc) - timedelta(days=settings.newsapi_days)
        ).strftime("%Y-%m-%d")

        params = {
            "q":        ticker,
            "from":     from_date,
            "sortBy":   "publishedAt",
            "pageSize": settings.newsapi_page_size,
            "language": "en",
            "apiKey":   settings.news_api_key,
        }
        url = f"{settings.newsapi_base_url}/everything"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        return [
            {
                "title":        a.get("title"),
                "description":  a.get("description"),
                "url":          a.get("url"),
                "source":       a.get("source", {}).get("name"),
                "published_at": a.get("publishedAt"),
            }
            for a in payload.get("articles", [])
        ]

    # ==================================================================
    # Options IV — nearest-expiry ATM strike
    # ==================================================================

    async def _get_iv(self, ticker: str) -> float | None:
        cached = await self._cache_get("iv_data", ticker)
        if cached is not None:
            return cached.get("iv") if isinstance(cached, dict) else cached
        loop = asyncio.get_event_loop()
        iv = await loop.run_in_executor(None, self._fetch_iv_sync, ticker)
        await self._cache_set("iv_data", ticker, {"iv": iv}, settings.iv_ttl)
        return iv

    @staticmethod
    def _fetch_iv_sync(ticker: str) -> float | None:
        """
        Fetch ATM implied volatility from the nearest-expiry options chain.

        Strategy
        --------
        1. Get current spot price.
        2. Pick the nearest expiry that has at least 5 days to expiration.
        3. From the call chain, find the strike closest to spot.
        4. Return that strike's ``impliedVolatility``.
        """
        try:
            tk = yf.Ticker(ticker)
            spot = tk.info.get("regularMarketPrice") or tk.info.get("currentPrice")
            if spot is None:
                hist = tk.history(period="2d")
                if hist.empty:
                    return None
                spot = float(hist["Close"].iloc[-1])

            expirations = tk.options
            if not expirations:
                return None

            # Pick first expiry with >= 5 calendar days out
            today = datetime.now(timezone.utc).date()
            chosen_expiry = None
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if (exp_date - today).days >= 5:
                    chosen_expiry = exp_str
                    break
            if chosen_expiry is None:
                chosen_expiry = expirations[0]

            chain = tk.option_chain(chosen_expiry)
            calls = chain.calls
            if calls is None or calls.empty:
                return None

            # ATM = strike closest to spot
            calls = calls[calls["impliedVolatility"] > 0].copy()
            if calls.empty:
                return None
            idx = (calls["strike"] - spot).abs().idxmin()
            iv = float(calls.loc[idx, "impliedVolatility"])
            return round(iv, 6)
        except Exception:
            return None

    # ==================================================================
    # SQLite cache helpers
    # ==================================================================

    def _require_db(self) -> None:
        if self._db is None:
            raise RuntimeError(
                "DataIngestion must be used as an async context manager."
            )

    async def _cache_get(
        self, table: str, ticker: str
    ) -> Any | None:
        """Return cached payload if a live row exists, else None."""
        now = _now()
        async with self._db.execute(
            f"SELECT data FROM {table} "
            f"WHERE ticker = ? AND ttl_expires_at > ? "
            f"ORDER BY fetched_at DESC LIMIT 1",
            (ticker, now),
        ) as cur:
            row = await cur.fetchone()
        return json.loads(row["data"]) if row else None

    async def _cache_set(
        self, table: str, ticker: str, data: Any, ttl: int
    ) -> None:
        """Upsert a cache row; prune expired rows for this ticker."""
        now        = _now()
        expires_at = now + ttl
        blob       = json.dumps(data)

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
# __main__ — smoke-test
# ---------------------------------------------------------------------------


async def _main() -> None:
    tickers = ["AAPL", "NVDA", "MSFT"]
    print(f"Testing DataIngestion with tickers: {tickers}\n")

    async with DataIngestion() as ing:
        data = await ing.get_stock_data(tickers)

    for ticker, d in data.items():
        print(f"{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        prices = d.get("prices", [])
        if prices:
            latest = prices[-1]
            print(f"  OHLCV rows   : {len(prices)}  (latest: {latest['date'][:10]} close={latest['close']})")
        else:
            print(f"  OHLCV        : {prices}")

        f = d.get("fundamentals", {})
        print(f"  P/E          : {f.get('pe_ratio')}")
        print(f"  EPS          : {f.get('eps')}")
        print(f"  Rev growth   : {f.get('revenue_growth')}")
        print(f"  Market cap   : {f.get('market_cap')}")

        news = d.get("news", [])
        print(f"  News         : {len(news)} articles{' (no key)' if not settings.news_api_key else ''}")

        print(f"  Options IV   : {d.get('options_iv')}")
        print()


if __name__ == "__main__":
    asyncio.run(_main())
