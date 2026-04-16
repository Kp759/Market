-- SQLite schema for the stock-screening system
-- All cache tables share (ticker, fetched_at, ttl_expires_at) contract.
-- Covering indexes on (ticker, ttl_expires_at) keep cache lookups O(log n).

-- ---------------------------------------------------------------------------
-- universe — deduplicated ticker list (~1800 symbols, 24-hr TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS universe (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    data           TEXT    NOT NULL,   -- JSON array of ticker strings
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);

-- ---------------------------------------------------------------------------
-- prices — 18-month daily OHLCV (15-min TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON array of OHLCV dicts
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS idx_prices_ttl    ON prices(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- fundamentals — P/E, EPS, revenue growth, market cap (15-min TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamentals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON object
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ttl    ON fundamentals(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- news_articles — NewsAPI articles (1-hr TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_articles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON array of article dicts
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_news_ttl    ON news_articles(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- iv_data — ATM implied volatility from options chain (1-hr TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS iv_data (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON: {"iv": 0.32}
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_iv_ticker ON iv_data(ticker);
CREATE INDEX IF NOT EXISTS idx_iv_ttl    ON iv_data(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- earnings — actual vs estimate, SUE components (6-hr TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS earnings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON: {actual, estimate, surprise, sue_score, quarterly_revenue:[...]}
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings(ticker);
CREATE INDEX IF NOT EXISTS idx_earnings_ttl    ON earnings(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- short_interest — days-to-cover, float short % from FINRA (6-hr TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS short_interest (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON: {short_interest, avg_volume, float_shares, days_to_cover, float_short_pct}
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_si_ticker ON short_interest(ticker);
CREATE INDEX IF NOT EXISTS idx_si_ttl    ON short_interest(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- insider_transactions — SEC EDGAR Form 4 (6-hr TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insider_transactions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON array of {date, insider, role, transaction_type, shares, value}
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_insider_ticker ON insider_transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_insider_ttl    ON insider_transactions(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- recommendations — LLM-generated buy/hold/sell output
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    score          REAL,
    recommendation TEXT,               -- 'buy' | 'hold' | 'sell'
    justification  TEXT,
    risk_flag      TEXT,
    position_size  REAL,               -- Kelly or equal-weight fraction
    signals        TEXT,               -- JSON blob of all input signals
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rec_ticker ON recommendations(ticker);
CREATE INDEX IF NOT EXISTS idx_rec_ttl    ON recommendations(ticker, ttl_expires_at);
CREATE INDEX IF NOT EXISTS idx_rec_score  ON recommendations(score DESC);
