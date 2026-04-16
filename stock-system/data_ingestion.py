"""
data_ingestion.py — async data-ingestion layer for the stock-screening system.

All yfinance calls run inside a Semaphore(30) with 3-attempt exponential
backoff starting at 2 s.  The entire module fails gracefully: individual
ticker errors are caught and logged; they never crash the pipeline.

Public methods on DataIngestion
--------------------------------
    get_ohlcv(tickers)                → {ticker: [OHLCV dicts]}  18-month daily
    get_fundamentals(tickers)         → {ticker: {P/E, EPS, rev-growth, …}}
    get_news(tickers)                 → {ticker: [article dicts]}  7-day window
    get_options_iv(tickers)           → {ticker: float | None}
    get_earnings(tickers)             → {ticker: {actual, estimate, quarterly_revenue}}
    get_short_interest(tickers)       → {ticker: {short_interest, avg_volume, float_shares}}
    get_insider_transactions(tickers) → {ticker: [Form-4 transaction dicts]}
    get_stock_data(tickers)           → unified dict combining all of the above

Cache TTLs (SQLite via aiosqlite)
----------------------------------
    prices / fundamentals : 15 min
    news / IV             : 1 hr
    earnings / SI / insider: 6 hr
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import aiosqlite
import httpx
import yfinance as yf

from config import settings

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "db" / "schema.sql"

# Process-global semaphore (created lazily inside an event loop)
_SEM: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(settings.semaphore_limit)
    return _SEM


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _with_retry(coro_fn, *args, label: str = "") -> Any:
    """
    Retry *coro_fn(*args)* up to ``settings.retry_attempts`` times using
    exponential back-off starting at ``settings.retry_backoff_base`` seconds.
    Returns ``None`` on total failure instead of propagating the exception.
    """
    backoff  = settings.retry_backoff_base
    last_exc: Exception | None = None
    for attempt in range(1, settings.retry_attempts + 1):
        try:
            return await coro_fn(*args)
        except Exception as exc:
            last_exc = exc
            if attempt < settings.retry_attempts:
                logger.debug(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    label, attempt, settings.retry_attempts, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
    logger.warning(
        "%s: all %d attempts failed — %s", label, settings.retry_attempts, last_exc
    )
    return None


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

async def _ensure_schema(db: aiosqlite.Connection) -> None:
    """Apply schema.sql idempotently (CREATE TABLE IF NOT EXISTS)."""
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
    Async context-manager for multi-source data ingestion.

    Usage::

        async with DataIngestion() as ing:
            data = await ing.get_stock_data(["AAPL", "NVDA", "MSFT"])

    Parameters
    ----------
    db_path : str, optional
        SQLite database path.  Defaults to ``settings.db_path``.
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
    # Unified entry-point
    # ==================================================================

    async def get_stock_data(self, tickers: list[str]) -> dict[str, Any]:
        """
        Fetch all data sources for every ticker in parallel.

        Returns
        -------
        dict
            ``{ticker: {prices, fundamentals, news, options_iv,
                        earnings, short_interest, insider_transactions}}``
        """
        self._require_db()
        results = await asyncio.gather(
            *[self._fetch_all(t) for t in tickers],
            return_exceptions=True,
        )
        return {
            t: r if not isinstance(r, Exception) else {"error": str(r)}
            for t, r in zip(tickers, results)
        }

    # ==================================================================
    # Individual public fetch methods
    # ==================================================================

    async def get_ohlcv(self, tickers: list[str]) -> dict[str, list[dict]]:
        """Return 18-month daily OHLCV per ticker."""
        self._require_db()
        results = await asyncio.gather(
            *[self._get_prices(t) for t in tickers], return_exceptions=True
        )
        return {t: (r if not isinstance(r, Exception) else []) for t, r in zip(tickers, results)}

    async def get_fundamentals(self, tickers: list[str]) -> dict[str, dict]:
        """Return fundamental ratios per ticker."""
        self._require_db()
        results = await asyncio.gather(
            *[self._get_fundamentals(t) for t in tickers], return_exceptions=True
        )
        return {t: (r if not isinstance(r, Exception) else {}) for t, r in zip(tickers, results)}

    async def get_news(self, tickers: list[str]) -> dict[str, list[dict]]:
        """Return last 7 days of news (empty list if NEWS_API_KEY missing)."""
        self._require_db()
        results = await asyncio.gather(
            *[self._get_news(t) for t in tickers], return_exceptions=True
        )
        return {t: (r if not isinstance(r, Exception) else []) for t, r in zip(tickers, results)}

    async def get_options_iv(self, tickers: list[str]) -> dict[str, float | None]:
        """Return ATM implied volatility from nearest-expiry options chain."""
        self._require_db()
        results = await asyncio.gather(
            *[self._get_iv(t) for t in tickers], return_exceptions=True
        )
        return {t: (r if not isinstance(r, Exception) else None) for t, r in zip(tickers, results)}

    async def get_earnings(self, tickers: list[str]) -> dict[str, dict]:
        """Return earnings actual/estimate, SUE components, quarterly revenue."""
        self._require_db()
        results = await asyncio.gather(
            *[self._get_earnings(t) for t in tickers], return_exceptions=True
        )
        return {t: (r if not isinstance(r, Exception) else {}) for t, r in zip(tickers, results)}

    async def get_short_interest(self, tickers: list[str]) -> dict[str, dict]:
        """Return short interest, days-to-cover, float short % from FINRA."""
        self._require_db()
        results = await asyncio.gather(
            *[self._get_short_interest(t) for t in tickers], return_exceptions=True
        )
        return {t: (r if not isinstance(r, Exception) else {}) for t, r in zip(tickers, results)}

    async def get_insider_transactions(self, tickers: list[str]) -> dict[str, list[dict]]:
        """Return SEC EDGAR Form-4 insider transactions (last 30 days)."""
        self._require_db()
        results = await asyncio.gather(
            *[self._get_insider(t) for t in tickers], return_exceptions=True
        )
        return {t: (r if not isinstance(r, Exception) else []) for t, r in zip(tickers, results)}

    # ==================================================================
    # Per-ticker orchestrator (private)
    # ==================================================================

    async def _fetch_all(self, ticker: str) -> dict[str, Any]:
        prices, fund, news, iv, earnings, si, insider = await asyncio.gather(
            self._get_prices(ticker),
            self._get_fundamentals(ticker),
            self._get_news(ticker),
            self._get_iv(ticker),
            self._get_earnings(ticker),
            self._get_short_interest(ticker),
            self._get_insider(ticker),
            return_exceptions=True,
        )

        def _safe(v: Any, default: Any) -> Any:
            return default if isinstance(v, Exception) else v

        return {
            "prices":               _safe(prices,  []),
            "fundamentals":         _safe(fund,    {}),
            "news":                 _safe(news,    []),
            "options_iv":           _safe(iv,      None),
            "earnings":             _safe(earnings,{}),
            "short_interest":       _safe(si,      {}),
            "insider_transactions": _safe(insider, []),
        }

    # ==================================================================
    # OHLCV — 18 months daily
    # ==================================================================

    async def _get_prices(self, ticker: str) -> list[dict[str, Any]]:
        cached = await self._cache_get("prices", ticker)
        if cached is not None:
            return cached

        sem  = _get_semaphore()
        loop = asyncio.get_event_loop()
        async with sem:
            data = await _with_retry(
                lambda t: loop.run_in_executor(None, self._fetch_prices_sync, t),
                ticker,
                label=f"prices:{ticker}",
            )
        if data is None:
            return []
        await self._cache_set("prices", ticker, data, settings.price_ttl)
        return data

    @staticmethod
    def _fetch_prices_sync(ticker: str) -> list[dict[str, Any]]:
        hist = yf.Ticker(ticker).history(period="18mo", interval="1d")
        records = []
        for ts, row in hist.iterrows():
            records.append({
                "date":   ts.isoformat(),
                "open":   round(float(row["Open"]),  4),
                "high":   round(float(row["High"]),  4),
                "low":    round(float(row["Low"]),   4),
                "close":  round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })
        return records

    # ==================================================================
    # Fundamentals
    # ==================================================================

    async def _get_fundamentals(self, ticker: str) -> dict[str, Any]:
        cached = await self._cache_get("fundamentals", ticker)
        if cached is not None:
            return cached

        sem  = _get_semaphore()
        loop = asyncio.get_event_loop()
        async with sem:
            data = await _with_retry(
                lambda t: loop.run_in_executor(None, self._fetch_fundamentals_sync, t),
                ticker,
                label=f"fundamentals:{ticker}",
            )
        if data is None:
            return {}
        await self._cache_set("fundamentals", ticker, data, settings.price_ttl)
        return data

    @staticmethod
    def _fetch_fundamentals_sync(ticker: str) -> dict[str, Any]:
        info = yf.Ticker(ticker).info or {}
        return {
            "pe_ratio":          info.get("trailingPE") or info.get("forwardPE"),
            "eps":               info.get("trailingEps") or info.get("forwardEps"),
            "eps_growth":        info.get("earningsGrowth"),
            "revenue":           info.get("totalRevenue"),
            "revenue_growth":    info.get("revenueGrowth") or info.get("earningsGrowth"),
            "market_cap":        info.get("marketCap"),
            "float_shares":      info.get("floatShares"),
            "shares_outstanding":info.get("sharesOutstanding"),
            "avg_volume":        info.get("averageVolume"),
            "beta":              info.get("beta"),
            "52w_high":          info.get("fiftyTwoWeekHigh"),
            "52w_low":           info.get("fiftyTwoWeekLow"),
            "sector":            info.get("sector"),
            "industry":          info.get("industry"),
        }

    # ==================================================================
    # News — last 7 days via NewsAPI (graceful skip when key absent)
    # ==================================================================

    async def _get_news(self, ticker: str) -> list[dict[str, Any]]:
        if not settings.news_api_key:
            return []   # no key → skip; sentiment defaults to 0.0

        cached = await self._cache_get("news_articles", ticker)
        if cached is not None:
            return cached

        articles = await _with_retry(
            self._fetch_news_async,
            ticker,
            label=f"news:{ticker}",
        )
        articles = articles or []
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
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{settings.newsapi_base_url}/everything", params=params
            )
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
    # Options IV — ATM, nearest expiry ≥ 5 days out
    # ==================================================================

    async def _get_iv(self, ticker: str) -> float | None:
        cached = await self._cache_get("iv_data", ticker)
        if cached is not None:
            return cached.get("iv") if isinstance(cached, dict) else cached

        sem  = _get_semaphore()
        loop = asyncio.get_event_loop()
        async with sem:
            iv = await _with_retry(
                lambda t: loop.run_in_executor(None, self._fetch_iv_sync, t),
                ticker,
                label=f"iv:{ticker}",
            )
        await self._cache_set("iv_data", ticker, {"iv": iv}, settings.iv_ttl)
        return iv

    @staticmethod
    def _fetch_iv_sync(ticker: str) -> float | None:
        try:
            tk   = yf.Ticker(ticker)
            info = tk.info or {}
            spot = info.get("regularMarketPrice") or info.get("currentPrice")
            if spot is None:
                hist = tk.history(period="2d")
                if hist.empty:
                    return None
                spot = float(hist["Close"].iloc[-1])

            expirations = tk.options
            if not expirations:
                return None

            today  = datetime.now(timezone.utc).date()
            chosen = None
            for exp_str in expirations:
                if (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days >= 5:
                    chosen = exp_str
                    break
            chosen = chosen or expirations[0]

            calls = tk.option_chain(chosen).calls
            if calls is None or calls.empty:
                return None
            calls = calls[calls["impliedVolatility"] > 0].copy()
            if calls.empty:
                return None
            idx = (calls["strike"] - spot).abs().idxmin()
            return round(float(calls.loc[idx, "impliedVolatility"]), 6)
        except Exception:
            return None

    # ==================================================================
    # Earnings — actual vs estimate + quarterly revenue
    # ==================================================================

    async def _get_earnings(self, ticker: str) -> dict[str, Any]:
        cached = await self._cache_get("earnings", ticker)
        if cached is not None:
            return cached

        sem  = _get_semaphore()
        loop = asyncio.get_event_loop()
        async with sem:
            data = await _with_retry(
                lambda t: loop.run_in_executor(None, self._fetch_earnings_sync, t),
                ticker,
                label=f"earnings:{ticker}",
            )
        data = data or {}
        await self._cache_set("earnings", ticker, data, settings.earnings_ttl)
        return data

    @staticmethod
    def _fetch_earnings_sync(ticker: str) -> dict[str, Any]:
        """
        Fetch most-recent EPS actual/estimate and last 8 quarterly revenues.
        Raw components are stored; signals.py computes the SUE score.
        """
        try:
            tk = yf.Ticker(ticker)

            actual  = None
            estimate = None
            hist_eps = tk.earnings_history
            if hist_eps is not None and not hist_eps.empty:
                latest   = hist_eps.iloc[-1]
                actual   = latest.get("epsActual")
                estimate = latest.get("epsEstimate")

            q_rev: list[float] = []
            try:
                qf = tk.quarterly_financials
                if qf is not None and not qf.empty:
                    for label in ("Total Revenue", "Revenue"):
                        if label in qf.index:
                            q_rev = [
                                float(v) for v in qf.loc[label].dropna().values[:8]
                            ]
                            break
            except Exception:
                pass

            return {
                "actual":            actual,
                "estimate":          estimate,
                "quarterly_revenue": q_rev,
            }
        except Exception as exc:
            logger.debug("earnings_sync %s: %s", ticker, exc)
            return {}

    # ==================================================================
    # Short interest — FINRA API + yfinance fallback
    # ==================================================================

    async def _get_short_interest(self, ticker: str) -> dict[str, Any]:
        cached = await self._cache_get("short_interest", ticker)
        if cached is not None:
            return cached

        data = await _with_retry(
            self._fetch_short_interest_async,
            ticker,
            label=f"si:{ticker}",
        )
        data = data or {}
        await self._cache_set("short_interest", ticker, data, settings.si_ttl)
        return data

    async def _fetch_short_interest_async(self, ticker: str) -> dict[str, Any]:
        si_shares: float | None = None
        try:
            params = {
                "limit":  1,
                "offset": 0,
                "fields": "symbolCode,shortInterestQty,settlementDate",
                "domainFilters": json.dumps([
                    {"fieldName": "symbolCode", "values": [ticker]}
                ]),
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(settings.finra_api_url, params=params)
                r.raise_for_status()
                records = r.json()
            if records:
                si_shares = float(records[0].get("shortInterestQty", 0) or 0)
        except Exception as exc:
            logger.debug("FINRA SI %s: %s", ticker, exc)

        # Supplement / fallback from yfinance info
        sem  = _get_semaphore()
        loop = asyncio.get_event_loop()
        async with sem:
            info = await _with_retry(
                lambda t: loop.run_in_executor(None, lambda: yf.Ticker(t).info or {}),
                ticker,
                label=f"si_info:{ticker}",
            ) or {}

        if si_shares is None:
            si_shares = info.get("sharesShort") or info.get("sharesShortPriorMonth")

        return {
            "short_interest": si_shares,
            "avg_volume":     info.get("averageVolume") or info.get("averageVolume10days"),
            "float_shares":   info.get("floatShares"),
        }

    # ==================================================================
    # Insider transactions — SEC EDGAR Form 4 ATOM feed
    # ==================================================================

    async def _get_insider(self, ticker: str) -> list[dict[str, Any]]:
        cached = await self._cache_get("insider_transactions", ticker)
        if cached is not None:
            return cached

        txns = await _with_retry(
            self._fetch_insider_async,
            ticker,
            label=f"insider:{ticker}",
        )
        txns = txns or []
        await self._cache_set(
            "insider_transactions", ticker, txns, settings.insider_ttl
        )
        return txns

    async def _fetch_insider_async(self, ticker: str) -> list[dict[str, Any]]:
        """
        Fetch recent Form-4 filings from SEC EDGAR ATOM feed.

        Returns transactions in the last 30 days with heuristically
        inferred insider role.  Transaction type and share counts
        require full XML parsing of the actual form; we store ``None``
        for those fields and let signals.py handle missing data.
        """
        params = {
            "action":   "getcompany",
            "CIK":      ticker,
            "type":     "4",
            "dateb":    "",
            "owner":    "include",
            "count":    40,
            "output":   "atom",
        }
        headers = {"User-Agent": settings.edgar_user_agent}

        try:
            async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
                r = await client.get(settings.edgar_rss_url, params=params)
                r.raise_for_status()
                raw_xml = r.text
        except Exception as exc:
            logger.debug("EDGAR RSS %s: %s", ticker, exc)
            return []

        cutoff       = datetime.now(timezone.utc) - timedelta(days=30)
        transactions: list[dict[str, Any]] = []

        try:
            root = ET.fromstring(raw_xml)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_el   = entry.find("atom:title", ns)
                updated_el = entry.find("atom:updated", ns)
                if title_el is None or updated_el is None:
                    continue
                try:
                    updated = datetime.fromisoformat(
                        updated_el.text.replace("Z", "+00:00")
                    )
                except Exception:
                    continue
                if updated < cutoff:
                    continue
                title        = (title_el.text or "").strip()
                insider_info = title.split(" - ", 1)[-1]
                transactions.append({
                    "date":             updated.isoformat(),
                    "insider":          insider_info,
                    "role":             _infer_role(insider_info),
                    "transaction_type": "unknown",
                    "shares":           None,
                    "value":            None,
                })
        except ET.ParseError as exc:
            logger.debug("EDGAR XML parse %s: %s", ticker, exc)

        return transactions

    # ==================================================================
    # SQLite cache helpers (private)
    # ==================================================================

    def _require_db(self) -> None:
        if self._db is None:
            raise RuntimeError(
                "DataIngestion must be used as an async context manager."
            )

    async def _cache_get(self, table: str, ticker: str) -> Any | None:
        """Return cached payload if a live (non-expired) row exists."""
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
        """Prune stale rows, then insert a fresh cache entry."""
        now        = _now()
        expires_at = now + ttl
        await self._db.execute(
            f"DELETE FROM {table} WHERE ticker = ? AND ttl_expires_at <= ?",
            (ticker, now),
        )
        await self._db.execute(
            f"INSERT INTO {table} (ticker, data, fetched_at, ttl_expires_at) "
            f"VALUES (?, ?, ?, ?)",
            (ticker, json.dumps(data), now, expires_at),
        )
        await self._db.commit()


# ---------------------------------------------------------------------------
# Role inference helper (module-level for clarity)
# ---------------------------------------------------------------------------

def _infer_role(insider_info: str) -> str:
    """Heuristically classify insider role from EDGAR filing description."""
    t = insider_info.lower()
    if "chief executive" in t or " ceo" in t:
        return "CEO"
    if "chief financial" in t or " cfo" in t:
        return "CFO"
    if "chief operating" in t or " coo" in t:
        return "COO"
    if "president" in t:
        return "President"
    if "director" in t:
        return "Director"
    if "officer" in t:
        return "Officer"
    return "Other"


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------

async def _main() -> None:
    tickers = ["AAPL", "NVDA", "MSFT"]
    print(f"DataIngestion smoke-test — tickers: {tickers}\n")

    async with DataIngestion() as ing:
        data = await ing.get_stock_data(tickers)

    for ticker, d in data.items():
        prices = d.get("prices", [])
        fund   = d.get("fundamentals", {})
        earn   = d.get("earnings", {})
        si     = d.get("short_interest", {})
        ins    = d.get("insider_transactions", [])
        print(f"{'='*55}  {ticker}")
        print(f"  OHLCV rows  : {len(prices)}")
        print(f"  P/E         : {fund.get('pe_ratio')}")
        print(f"  Options IV  : {d.get('options_iv')}")
        print(f"  News        : {len(d.get('news', []))} articles")
        print(f"  EPS act/est : {earn.get('actual')} / {earn.get('estimate')}")
        print(f"  Q-revenues  : {len(earn.get('quarterly_revenue', []))} quarters")
        print(f"  Short int   : {si.get('short_interest')}")
        print(f"  Insider tx  : {len(ins)} (last 30 days)")
        print()


if __name__ == "__main__":
    asyncio.run(_main())
